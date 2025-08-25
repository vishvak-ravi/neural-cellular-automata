// PokemonConfigModel.js - Manages pokemon configuration and validation
import { Constants } from '../utils/Constants.js';

export class PokemonConfigModel {
  constructor() {
    this.pokemonList = Constants.POKEMON;
    this.pokemonSizes = Constants.POKEMON_SIZES;
    this.pokemonChannels = Constants.POKEMON_CHANNELS;
    this.unsupportedCombos = new Set(Constants.UNSUPPORTED_COMBOS);
    this.modes = Constants.MODES;
  }

  /**
   * Get list of available pokemon
   */
  getPokemonList() {
    return [...this.pokemonList];
  }

  /**
   * Get configuration for a specific pokemon
   */
  getPokemonConfig(name) {
    if (!this.pokemonList.includes(name)) {
      return null;
    }
    
    return {
      name,
      size: this.pokemonSizes[name],
      channelCount: this.pokemonChannels[name]
    };
  }

  /**
   * Get pokemon size
   */
  getSize(name) {
    return this.pokemonSizes[name] || null;
  }

  /**
   * Get pokemon channel count
   */
  getChannelCount(name) {
    return this.pokemonChannels[name] || null;
  }

  /**
   * Check if a pokemon-mode combination is valid
   */
  isValidCombination(pokemon, mode) {
    const combo = `${pokemon}_${mode}`;
    return !this.unsupportedCombos.has(combo);
  }

  /**
   * Get available modes
   */
  getModes() {
    return [...this.modes];
  }

  /**
   * Get available modes for a specific pokemon
   */
  getAvailableModesForPokemon(pokemon) {
    return this.modes.filter(mode => this.isValidCombination(pokemon, mode));
  }
}
