// PokemonSelectorView.js - Handles pokemon selection UI
import { View } from './View.js';

export class PokemonSelectorView extends View {
  constructor(element, eventBus) {
    super(element, eventBus);
    this.selectedPokemon = null;
  }

  /**
   * Render the pokemon grid
   */
  render(pokemonList) {
    // Clear existing content
    this.element.innerHTML = '';
    
    // Create pokemon items
    pokemonList.forEach(pokemon => {
      const div = document.createElement('div');
      div.className = 'subject';
      div.id = pokemon;
      
      const img = document.createElement('img');
      img.src = `pokemon/${pokemon}.png`;
      img.alt = pokemon;
      img.width = 50;
      img.height = 50;
      
      div.appendChild(img);
      this.element.appendChild(div);
    });
    
    // Bind events
    this.bindEvents();
    
    // Set initial selection if exists
    if (this.selectedPokemon) {
      this.setSelectedPokemon(this.selectedPokemon);
    }
  }

  /**
   * Bind click events to pokemon items
   */
  bindEvents() {
    // Unbind previous events
    this.unbindEvents();
    
    // Bind click event to each pokemon
    const pokemonElements = this.element.querySelectorAll('.subject');
    pokemonElements.forEach(element => {
      const handler = () => {
        const pokemonName = element.id;
        this.emit('selector:pokemon-clicked', { pokemon: pokemonName });
      };
      
      element.addEventListener('click', handler);
      this.eventHandlers.set(`#${element.id}:click`, handler);
    });
  }

  /**
   * Set the selected pokemon
   */
  setSelectedPokemon(name) {
    // Remove previous selection
    const previousSelected = this.element.querySelector('.subject-selected');
    if (previousSelected) {
      previousSelected.classList.remove('subject-selected');
    }
    
    // Add new selection
    const newSelected = this.element.querySelector(`#${name}`);
    if (newSelected) {
      newSelected.classList.add('subject-selected');
      this.selectedPokemon = name;
    }
  }

  /**
   * Enable a pokemon (remove disabled state)
   */
  enablePokemon(name) {
    const element = this.element.querySelector(`#${name}`);
    if (element) {
      element.classList.remove('disabled');
      element.style.opacity = '1';
      element.style.pointerEvents = 'auto';
    }
  }

  /**
   * Disable a pokemon
   */
  disablePokemon(name) {
    const element = this.element.querySelector(`#${name}`);
    if (element) {
      element.classList.add('disabled');
      element.style.opacity = '0.5';
      element.style.pointerEvents = 'none';
    }
  }

  /**
   * Update available pokemon based on current mode
   */
  updateAvailability(availabilityMap) {
    Object.entries(availabilityMap).forEach(([pokemon, isAvailable]) => {
      if (isAvailable) {
        this.enablePokemon(pokemon);
      } else {
        this.disablePokemon(pokemon);
      }
    });
  }
}
